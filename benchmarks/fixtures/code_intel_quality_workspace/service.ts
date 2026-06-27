export interface UserRecord {
  id: string;
}

export function loadUser(id: string): UserRecord {
  return { id };
}

export function buildSession(userId: string) {
  return loadUser(userId);
}
